"""
The following is a simple example algorithm.

It is meant to run within a container.

To run the container locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

from pathlib import Path
import json


# INPUT_PATH = Path("/input")
INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def run():
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()
    print("Inputs: ", interface_key)
    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        ("endoscopic-robotic-surgery-video",  "visual-context-question",): interf0_handler,}[interface_key]

    # Call the handler
    return handler()


def interf0_handler():
    # Read the input
    # input_endoscopic_robotic_surgery_video = load_file(
    #     location=INPUT_PATH / "endoscopic-robotic-surgery-video.mp4",
    # )
    input_visual_context_question = load_json_file(
        location=INPUT_PATH / "visual-context-question.json",
    )
    print('Question: ', json.dumps(input_visual_context_question, indent=4))


    # Process the inputs: any way you'd like
    #_show_torch_cuda_info()

    # Some additional resources might be required, include these in one of two ways.

    # Option 1: part of the Docker-container image: resources/
    # resource_dir = Path("/opt/app/resources")
    # with open(resource_dir / "some_resource.txt", "r") as f:
    #     print(f.read())

    # # Option 2: upload them as a separate tarball to Grand Challenge (go to your Algorithm > Models). The resources in the tarball will be extracted to `model_dir` at runtime.
    # model_dir = Path("/opt/ml/model")
    # with open(
    #     model_dir / "a_tarball_subdirectory" / "some_tarball_resource.txt", "r"
    # ) as f:
    #     print(f.read())

    # For now, let us make bogus predictions
    output_visual_context_response = "Example String"
    print("Output: ", output_visual_context_response)
    # Save your output
    
    write_json_file(
        location=OUTPUT_PATH / "visual-context-response.json",
        content=output_visual_context_response,
    )
    print('output saved to  ', OUTPUT_PATH)

    return 0


def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    print('These are the inputs:' , inputs)
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


# Note to the developer:
#   the following function is very generic and should likely
#   be adopted to something more specific for your algorithm/challenge
def load_file(*, location):
    # Reads the content of a file
    with open(location) as f:
        return f.read()


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
