from mmseg.apis import inference_segmentor, init_segmentor
import torch
import mmcv
import cv2
import os 
import glob
import time

def list_available_gpus():
    available_gpus = []
    for i in range(torch.cuda.device_count()):
        available_gpus.append(torch.cuda.get_device_name(i))
    return available_gpus


def process_videos(config_file, checkpoint_file, videos_folder, destination_folder, cuda_gpu):
    print(config_file)
    print(checkpoint_file)
    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

    # Use glob to get a list of all MP4 files in the directory
    mp4_files = glob.glob(os.path.join(videos_folder, '*.mp4'))

    # Iterate through the list of MP4 files
    for mp4_file in mp4_files:
        start_time = time.time()

        mp4_filename = os.path.basename(mp4_file)
        print("analyzing video" + mp4_file)

        new_destination_path = os.path.join(destination_folder, mp4_filename)

        video_reader = mmcv.VideoReader(mp4_file)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        Video_writer = cv2.VideoWriter(
                new_destination_path, fourcc, video_reader.fps,
                (video_reader.width, video_reader.height))
        
        for frame in video_reader:
            result = inference_segmentor(model, frame)
            frame = model.show_result(frame, result, wait_time=1)
            Video_writer.write(frame)

        # Record the end time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Video analysis time: " + str(elapsed_time))


        Video_writer.release()



def main():
    gpus = list_available_gpus()
    print(gpus) 
    checkpoints_folder = 'checkpoints\\'

    # Directory containing the MP4 files
    videos_directory = 'videos\\'
    destination_directory = 'analyzed\\'

    items = os.listdir(checkpoints_folder)
    folders = [item for item in items if os.path.isdir(os.path.join(checkpoints_folder, item))]

    for folder in folders:
        folder_path = os.path.join(checkpoints_folder, folder)
        files_in_folder = os.listdir(folder_path)
        print(folder)

        # Check for .py and .pth files and print their names
        py_files = [file for file in files_in_folder if file.endswith('.py')]
        pth_files = [file for file in files_in_folder if file.endswith('.pth')]

        if py_files and pth_files:
            if len(py_files) and len(pth_files):
                for py_file, pth_file in zip(py_files, pth_files):
                    config_file = os.path.join(folder_path, py_file)
                    checkpoint_file = os.path.join(folder_path, pth_file)
                    #print(f'Processing pair: {config_file}, {checkpoint_file}')
                    new_directory = os.path.join(destination_directory, folder)
                    if not os.path.exists(new_directory):
                        os.mkdir(new_directory)
                    
                    process_videos(config_file, checkpoint_file, videos_directory, new_directory, gpus[0])
               


if __name__ == "__main__":
    main()