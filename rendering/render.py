import os
import time
import yaml
from joblib import Parallel, delayed
import glob


def render(cnt, model, params):
    name = model.replace(".stl", "_render.yml")
    path = "/".join(model.split("/")[:-1])
    params["model"] = model
    params["count"] = cnt
    params["folder"] = path
    with open(name, mode="w") as fp:
        yaml.dump(params, fp, allow_unicode=True)
    print("Rendering model %i with path %s." % (cnt, model))
    dir_path = os.path.dirname(os.path.realpath(__file__))

    command = "blender %s/resources/scene.blend --background --python %s/render_object.py -- %s"
    command = command % (dir_path, dir_path, name) + " > /dev/null"

    # print(command)
    start_obj = time.time()
    os.system(command)
    print("{} RENDERING TIME: {}".format(model, time.time() - start_obj))
    return


def render_batch(model_list, parameters):
    # Update parameters
    jobs = parameters["render_threads"]
    Parallel(n_jobs=jobs)(delayed(render)(cnt, model, parameters) for cnt, model in model_list)
    return


def print_names(model_list):
    with open('./images/names.txt', 'w') as file:
        for i, x in model_list:
            file.write('%02d: %s\n' % (i, x.split('/')[-1]))
    return


if __name__ == "__main__":
    config_path = "resources/render_config.yml"
    with open(config_path) as f:
        arguments = yaml.safe_load(f)
        arguments["path"] = arguments["path"] % os.path.dirname(os.path.realpath(__file__))

    models = glob.glob(os.path.join(arguments["path"], "*.stl"))
    models = list(enumerate(sorted(models)))

    start = time.time()
    render_batch(models[:], arguments)
    print("TOTAL RENDERING TIME: {}".format(time.time() - start))
    print_names(models)
