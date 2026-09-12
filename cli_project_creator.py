import os
import shutil
from pprint import pprint
import argparse
import re

def subfolders_from_files(files):
    files_ = [os.path.split(i)[-1] for i in files]
    file_types = set([i.split('.')[-1] for i in files])
    all_cleaned = []
    for file_type in file_types:
        selects = [i for i in files_ if i.endswith(file_type)]
        cleaned = []
        for file in selects:
            file = file.split('.' + file_type)[0]
            if 'DLC' in file:
                file = file.split('DLC')[0]
            if 'frame_synced' in file:
                file = file.split('-frame_synced')[0]
            if re.match('(.+)cam[0-5]\Z', file): # Ideally, original video files were named like date-animal_cam1.avi
                file = re.match('(.+)cam[0-5]\Z', file)[1]
            if re.match('(.+)cam[0-5](.+)', file): # Some may unfortunately be named date_cam1_animal.avi
                pre = re.match('(.+)cam[0-5](.+)', file)[1]
                suf = re.match('(.+)cam[0-5](.+)', file)[2]
                file = ''.join((pre,suf))
            if re.match('cam[0-5](.+)', file): # This should never happen
                file = re.match('cam[0-5](.+)', file)[1]
            if file.endswith(('-','_')):
                file = file[:-1]
            if file.startswith(('-', '_')): # This should also never happen
                file = file[1:]
            cleaned.append(file)
        all_cleaned.append(cleaned)
    all_cleaned = [item for subv in all_cleaned for item in subv]
    all_cleaned = list(set(all_cleaned))
    date_structure = '([0-9]+-[0-9]+-[0-9]+)'
    secondary_date_structure = '([0-9]{4,6})'
    if all([re.match(date_structure, i) for i in all_cleaned]):
        project_name = re.match(date_structure, all_cleaned[0])[1]
    elif all([re.match(secondary_date_structure, i) for i in all_cleaned]):
        project_name = re.match(secondary_date_structure, all_cleaned[0])[1]
    else:
        for i in range(len(all_cleaned[0])):
            if len(set([name[i] for name in all_cleaned])) == 1:
                pass
            else:
                project_name = all_cleaned[0][:i]
                break
    return all_cleaned, project_name

def create_folder_structure(root_dir, project_dir, subdirs: list = None):
    anipose_folders = ['calibration', 'pose-2d']
    project_root = os.path.join(root_dir, project_dir)
    assert not os.path.isdir(project_root), f'Folder already exists at {project_root}!'
    for subdir in subdirs:
        dir_ = os.path.join(project_root, subdir)
        for anipose_folder in anipose_folders:
            os.makedirs(os.path.join(dir_, anipose_folder))
    run_from = os.path.split(__file__)[0]
    config_file = os.path.join(run_from, 'config.toml')
    assert os.path.isfile(config_file), f'Could not find a config.toml file in {run_from}'
    shutil.copy2(config_file, project_root)

def create_structure(video_dir, target_dir):
	assert os.path.isdir(target_dir), 'Please use the full path to the new project target directory'
	videos = [i for i in os.listdir(video_dir) if i.endswith('avi')]
	assert len(videos) > 0, f'No videos found in {video_dir}. Did you use a full path?'
	sub_p_dirs, p_dir = subfolders_from_files(videos)
	sub_p_dirs = [i for i in sub_p_dirs if 'calib' not in i]
	root_dir = target_dir # Project destination
	visualize = {
		root_dir : {
			p_dir : [
				[i for i in sub_p_dirs]
			]
		}
	}
	print(f'\n\nProject structure:')
	pprint(visualize)
	while True:
		goahead = input('(Y/N) Is the above structure correct?\n>')
		if goahead.lower() in ('n', 'no', '0'):
			return
		if goahead.lower() in ('y', 'yes', '1'):
			create_folder_structure(root_dir, p_dir, sub_p_dirs)
			return
		print(f'Sorry, please enter either y/n, yes/no, or 0/1')

def populate_folder(source_dir, target_dir_):
	source_files = [i for i in os.listdir(source_dir) if i.endswith(('.csv', '.h5'))]
	source_files = [i for i in source_files if not i.startswith('.')]

	target_dirs = [i for i in os.listdir(target_dir_) if os.path.isdir(os.path.join(target_dir_, i))]
	target_dirs.sort(key = lambda x: len(x), reverse = True)

	planned_copy = []
	planned_copy_ = []

	for target_dir in target_dirs:
		associated_files = [i for i in source_files if target_dir in i]
		source_files = [i for i in source_files if not i in associated_files]
		associated_files = [i for i in associated_files if 'frame_synced' in i]
		#print(f'This is the target: {target_dir}\nThese are the associated files:\n{associated_files}')
		planned_copy.append((target_dir, associated_files))
		planned_copy_.append((target_dir, len(associated_files)))

	plan = {t_dir : t_files for (t_dir, t_files) in planned_copy_}
	print(f'Target folders and the number of files headed that way:')
	pprint(plan)
	while True:
		goahead = input('(Y/N) Would you like to copy files with above plan?\n>')
		if goahead.lower() in ('n', 'no', '0'):
			return
		if goahead.lower() in ('y', 'yes', '1'):
			for target, associated in planned_copy:
				target_path = os.path.join(target_dir_, target, 'pose-2d')
				file_paths = [os.path.join(source_dir, i) for i in associated]
				for file_path in file_paths:
					shutil.copy(file_path, target_path)
			return
		print(f'Sorry, please enter either y/n, yes/no, or 0/1\n>')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--source', required=True, help='Path containing all synced videos and DLC tracking files')
	parser.add_argument('--project', required=True, help='Path to output project folder')
	args = parser.parse_args()

	create_structure(args.source, args.project)
	populate_folder(args.source, args.project)

