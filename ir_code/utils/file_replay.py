'''
仿真回放-文件拆分utils
'''
import os

def replay_file(output_dir):
    file_name=output_dir+'xuancheng_test.log'
    target_size=0.4* 1024 * 1024 * 1024
    target_file_prefix = 'replay'
    counter = 0
    target_file_num = 0
    with open(file_name, 'r') as source_file:
        target_file_name = f'{target_file_prefix}_{target_file_num}.txt'
        target_file = open(output_dir+target_file_name, 'w')
        for line in source_file:
            target_file.write(line)
            counter+=1
            if os.path.getsize(output_dir+target_file_name) >= target_size:
                target_file.close()
                target_file_num += counter
                counter = 0
                target_file_name = f'{target_file_prefix}_{target_file_num}.txt'
                target_file = open(output_dir+target_file_name, 'w')
        target_file.close()

output_dir ='./log/'
replay_file(output_dir)
