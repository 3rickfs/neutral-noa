import os
import json
import requests
import ctypes
import time

from flask import Flask, request

import boto3

from orchestration_planner import (#read_endpoints,
                                   OrchPlannerOps,
                                   save_files,
                                   read_json_data
                                  )
from synapses import synapses_process
from neuron_distributor import (start_distribution,
                                start_first_layer_input_distribution,
                                send_order_to_nods_to_delete_sp
                               )
from noaDBmanager import noaDBmanager

app = Flask(__name__)
syn_proc = None
#home = os.environ.get('HOME')
#if 'dev-1' in home:
#    app.config['UPLOAD_FOLDER'] = '/home/dev-1/dev/edge-intelligence-' + \
#                                  'simulator/neuro-orchestrator-agent/uploads'
#else:
#    app.config['UPLOAD_FOLDER'] = '/home/noa/neuro-orchestrator-agent/uploads'

app.config['UPLOAD_FOLDER'] = os.getcwd() + "/uploads"

#Create/override synapses processes json file to be empty
# means no synaptic process on RAM
empty_json = {}
with open(app.config['UPLOAD_FOLDER'] + "/synapses_processes.json", "w") as jf:
    json.dump(empty_json, jf)
jf.close()

def delete_sp_obj(syn_proc, synapses_process_id):
    #Delete json file
    os.remove(syn_proc.obj_local_path)

    #Delete S3 obj
    syn_proc.delete_synproc_aimodel_s3_objs()

    #Delete registers
    p = app.config['UPLOAD_FOLDER']
    with open(p + "/synapses_processes.json", "r") as jf:
        sps = json.load(jf)
    jf.close()
    del syn_proc, sps[str(synapses_process_id)]

    #Overriding local registers
    with open(p + "synapses_processes.json", "w") as jf:
        json.dump(sps, jf)
    jf.close()

    #Delete register in remote db
    noaDBmanager.delete_synproc_register(synapses_process_id)

def get_synapses_code(spid, uid):
    global syn_proc

    print("Getting synapses code")
    p = app.config["UPLOAD_FOLDER"]
    with open(p + "/synapses_processes.json", "r") as jf:
        synapses_processes = json.load(jf)
    jf.close()

    spc = 0
    #try:
    #    print(f"sp: {synapses_processes}")
    #    spc = synapses_processes[str(spid)]
    #    print("correct spc: {spc}")

    #    syn_proc = ctypes.cast(
    #        int(spc),
    #        ctypes.py_object
    #    ).value
    #    print(f"syn proc: {syn_proc}")
    #    ma = syn_proc.mem_adr()
    #    print(f"memory address: {ma}")

    #except:

    # means sp isn't in RAM, then it should be loaded from disk
    pspfn = "persistent_synapses_processes.json"
    with open(p + "/" + pspfn, "r") as jf:
        sps = json.load(jf)
    jf.close()

    try:
        spc = sps[str(spid)]
        fp = os.listdir(p + "/sps")
        files = [
            f for f in fp if int(f.split("-")[0]) == int(uid)
        ]

        #files = []
        #for f in fp:
        #    ruid = int(f.split("-")[0])
        #    if ruid == int(uid):  #os.path.isfile(f) and 
        #        files.append(f)
        #    else:
        #        print(f"For file {f}, ruid: {ruid} is not iqual to uid: {uid}")

        for f in files:
            print(f"f: {f}, spc: {spc}")
            if str(spc) in f:
                #then we need to load the sp
                with open(p + "/sps/" + f, "r") as jf:
                    sp = json.load(jf)
                jf.close()

                # reload sp
                syn_proc = synapses_process()
                syn_proc.reload_synaptic_process(sp)
                print(syn_proc)
                fleps = syn_proc.read_fleps()
                print(f"fleps: {fleps}")
                print("lala")

                nspc = id(syn_proc) #new syn proc code
                synapses_processes[str(spid)] = nspc
                sps[str(spid)] = nspc
                spc = nspc
                #update sp json file name 
                sfn = f.split("-")
                sfn[1] = str(nspc)
                nfn = ""
                for i in sfn:
                    nfn += i + "-"
                nfn = nfn[:-1]
                nfp = p + '/sps/' + nfn
                os.rename(p + '/sps/' + f, nfp)
                #Update the local object path
                syn_proc.obj_local_path = nfp

                #TODO: Update sp and aimodel json file name in cloud
                #seems like it's not necessary
                print("lalala")

                print(f"New synapses_processes: {synapses_processes}")
                print(f"New persistent sp: {sps}")

                with open(p + "/synapses_processes.json", "w") as jf:
                    json.dump(synapses_processes, jf)
                jf.close()

                with open(p + "/" + pspfn, "w") as jf:
                    json.dump(sps, jf)
                jf.close()
                break
    except Exception as e:
        print(f"Synaptic process error: {e}")
        spc = 0

    return spc

def get_synapses_obj_memory_address(synapses_process_id):
    print("Getting synapses object memory address")
    with open("synapses_processes.json", "r") as jsonfile:
        synapses_processes = json.load(jsonfile)
    jsonfile.close()

    return synapses_processes[str(synapses_process_id)]

def get_fleps(nod_info):
    print("Getting first layer endpoints")
    fleps = []
    #print(f"nod info: {nod_info}")
    for nodi in nod_info:
        cid = nod_info[nodi]["capa_ids"][0]
        if cid == 1: #first layer only
            print(f"nod first layer endpoint: {(nod_info[nodi]['ops_ep'])}")
            fleps.append(nod_info[nodi]["ops_ep"])
        else:
            break

    return fleps

@app.route("/")
def about():
    return "<p>Neuro orchestrator agent"+ \
           " - developed with love by Tekvot dev team. </p>"

@app.route("/distribute_neurons", methods = ['POST'])
def distribute_neurons():
    nod_dict = {}
    #json_data = request.get_json()

    file_path = app.config['UPLOAD_FOLDER']
    #js_n, ops_n, dis_n = save_files(request.files.getlist('files'), file_path)
    js_n = save_files(request, file_path)
    json_data = read_json_data(file_path, js_n)

    #these EPs are to share ops info btw NODs
    #nod_ops_ep = read_endpoints("ops_eps")
    #Need other EP to send neuorns info from NO to NODs
    #nod_dis_ep = read_endpoints("dis_eps")

    synapses_process_id = json_data["synapses_process_id"]
    #to save info about first layer endpoint to be used later
    syn_proc = ctypes.cast(
        int(synapses_process_id),
        ctypes.py_object
    ).value
    #this will be used for the neuro orchestrator to send json to nods too
    #nod_ep = read_endpoints("./nod_endpoints.txt")
    neuro_orchestrator_ep = [json_data["neuro_orchestrator_ep"]]

    #Orchestration planning
    try:
        nod_dict = OrchPlannerOps.run(
            #nod_ep = nod_ops_ep,
            nods_tech_info = syn_proc.nods_tech_info,
            neuro_orchestrator_ep = neuro_orchestrator_ep,
            json_data = json_data
        )["nod_dict"]
    except Exception as e:
        print(f"error during orchestration planning: {e}")

    #Getting first layer endpoints and save them into synapses process obj
    fl_eps = get_fleps(nod_dict)
    syn_proc.save_fleps(fl_eps)
    #Distribution of neurons
    try:
        nod_res = start_distribution(nod_dict,
                                     #nod_dis_ep,
                                     synapses_process_id
                                    )
    except Exception as e:
        print(f"error during orchestration planning: {e}")

    return json.dumps(str(nod_res))

@app.route("/start_synapses_process", methods=['POST'])
def start_synapses_process():
    global syn_proc

    input_data = request.get_json()
    syn_proc = synapses_process(**input_data)

    synapses_process_id = id(syn_proc)
    #syn_proc.set_object_memory_address(synapses_process_id)
    output = {"synapses_process_id": synapses_process_id}

    return json.dumps(output)

@app.route("/send_inputs_to_1layer_nods", methods=['POST'])
def send_inputs_to_1layer_nods():
    global syn_proc

    input_data = request.get_json()
    user_id = input_data["user_id"]
    username = input_data["username"]
    if noaDBmanager.verify_user_compliance(
        user_id,
        username
    ):
        del input_data["user_id"], input_data["username"]

        sp_c = get_synapses_code(input_data["synapses_process_id"], user_id)
        print(f"sp_c: {sp_c}")
        if sp_c != 0:
            syn_proc = ctypes.cast(
                #int(input_data["synapses_process_id"]),
                int(sp_c),
                ctypes.py_object
            ).value
            print(syn_proc)

            nod_eps = syn_proc.read_fleps()
            #get time to calculate performance
            t = time.time()
            syn_proc.set_pred_start_time(t)
            #distribute inputs to nods
            result = start_first_layer_input_distribution(input_data,
                                                          nod_eps)
            result = {"result": result}
        else:
            result = {"result": "Error, synaptic process does not exist"}
    else:
        result = {"result": "Error - user not available"}

    return json.dumps(result)

@app.route("/set_final_output", methods=['POST'])
def set_final_output():
    global syn_proc 

    input_data = request.get_json()
    #syn_proc_id = get_synapses_obj_memory_address(
    #    input_data["synapses_process_id"]
    #)
    user_id = 0 #dummy value on this case only
    sp_c = get_synapses_code(input_data["synapses_process_id"], user_id)
    print(f"prediction result: {input_data['inputs']}")
    print(f"sp id: {input_data['synapses_process_id']}")
    print(f"synaptic code: {sp_c}")

    if sp_c != 0:
        try:
            syn_proc = ctypes.cast(
                #int(input_data["synapses_process_id"]),
                sp_c,
                ctypes.py_object
            ).value

            print(f"saved start time: {syn_proc.get_pred_start_time()}")
            syn_proc.set_synapses_output(input_data["inputs"])
            #get end time to calculate performance
            t = time.time()
            syn_proc.set_pred_end_time(t)
            #print(t)
            #calculate prediction time
            syn_proc.calculate_prediction_time()
            #Saving modifications to obj
            syn_proc.export_obj_as_json()
            result = {"result": "ok"}
        except:
            result = {"result": "error"}
    else:
        result = {"result": "error"}

    return result #"ok"

@app.route("/read_synapses_process_output", methods=['POST'])
def read_synapses_process_output():
    global syn_proc

    input_data = request.get_json()
    user_id = input_data["user_id"]
    username = input_data["username"]
    if noaDBmanager.verify_user_compliance(
        user_id,
        username
    ):
        sp_c = get_synapses_code(input_data["synapses_process_id"], user_id)
        print(f"sp id: {input_data['synapses_process_id']}")
        print(f"synaptic code: {sp_c}")
        syn_proc = ctypes.cast(
            #int(input_data["synapses_process_id"]),
            sp_c,
            ctypes.py_object
        ).value
        synapses_output = syn_proc.read_synapses_output()
        res = {
            "synapses_output": synapses_output,
            "pred_time": syn_proc.get_prediction_time(),
            "power_consumption": syn_proc.calculate_pred_power_consumption(),
            "carbon_footprint": syn_proc.calculate_carbon_footprint(),
        }
        #Saving modifications to obj
        syn_proc.export_obj_as_json()
    else:
        res = {"result": "Error - user is not available"}

    return res

@app.route("/create_user", methods=['POST'])
def create_user():
    input_data = request.get_json()

    #Create a new user
    try:
        res = noaDBmanager.create_new_user(input_data)
    except Exception as e:
        print(f"There was an error creating a new user: {e}")
        res = {"error": e}

    #Returning results: username & user_id
    return res

@app.route("/crear_proceso_sinaptico", methods=['POST'])
def crear_proceso_sinaptico():
    global syn_proc

    file_path = app.config['UPLOAD_FOLDER']
    js_n, ni_n = save_files(request, file_path)
    json_data = read_json_data(file_path, js_n)
    nods_info = read_json_data(file_path, ni_n)

    #input_data = request.get_json()
    syn_proc = synapses_process(nods_info)

    synapses_process_id = id(syn_proc)

    # NOA endpoint for getting model output
    neuro_orchestrator_ep = [json_data["neuro_orchestrator_url"] + "/set_final_output"]

    #Onboard the model
    data_4_onboarding = {}
    data_4_onboarding["spcode"] = synapses_process_id
    data_4_onboarding["noep"] = neuro_orchestrator_ep
    #data_4_onboarding["ni"] = nods_info
    data_4_onboarding["upload_folder_path"] = file_path
    data_4_onboarding["user_id"] = json_data["user_id"]
    data_4_onboarding["username"] = json_data["username"]
    data_4_onboarding["mj"] = json_data #here comes the model.json too
    data_4_onboarding["sc_fpath"] = "synaptic_process_objs"
    data_4_onboarding["dataset_name"] = json_data["dataset_name"]
    data_4_onboarding["dataset_url"] = json_data["dataset_url"]
    data_4_onboarding["notebook_url"] = json_data["notebook_url"]
    #data for the model
    data_4_onboarding["mfpc"] = "models"
    #data_4_onboarding["mfpl"] = app.config['UPLOAD_FOLDER']
    data_4_onboarding["model_bucket_name"] = "greenbrain"

    try:
        res = syn_proc.onboard_model(**data_4_onboarding)
            #spid = synapses_process_id,
            #noep = neuro_orchestrator_ep,
            #upload_folder_path = file_path,
            #user_id = json_data["user_id"],
            #username = json_data["username"],
            #mj = json_data, #here comes the model.json too
            #sc_fpath = "synaptic_process_objs",
            #dataset_name = json_data["dataset_name"],
            #dataset_url = json_data["dataset_url"],
            #notebook_url = json_data["notebook_url"],
            #mfpc = "models",
            #model_bucket_name = "greenbrain"
        #)
    except Exception as e:
        print(f"Error when creating synaptic process: {e}")
        res = {"res":f"error - {e}"}

    return json.dumps(res) #str(res))

@app.route("/delete_proceso_sinaptico", methods=['POST'])
def delete_proceso_sinaptico():
    json_data = request.get_json()
    synapses_process_id = json_data["synapses_process_id"]
    syn_proc = ctypes.cast(
        int(synapses_process_id),
        ctypes.py_object
    ).value

    try:
        print("Veriying user availability")
        if noaDBmanager.verify_user_compliance(
            json_data["user_id"],
            json_data["username"]
        ):
            send_order_to_nods_to_delete_sp(syn_proc)
            delete_sp_obj(syn_proc, synapses_process_id)
            res = {"result": "ok"}
        else:
            res = {"result": "Error, user id or username are not valid"}


    except Exception as e:
        print(f"Error deleting synaptic process obj: {e}")
        res = {"result": "Error: {e}"}

    return res

if __name__ == '__main__':
    app.run(host=host, port=int(port))
