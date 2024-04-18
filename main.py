import argparse
import subprocess
from RealTimeELGandVGE import real_time_ELG_VGE as RTEV
from RealTimeVGE import real_time_VGE as RTV
from VideoELGandVGE import video_ELG_VGE as VEV
from VideoVGE import video_VGE as VV
from useServer import run_program


'''
命令行代码解释：
--IsUseELG：是否使用ELG模型估计瞳孔中心，1为使用，0为不使用
--IsRealTime：是否实时处理视频流，1为实时处理，0为先录制视频再处理
--modelPath：模型路径，如./model256_20.pth
--IsUseServer：是否使用服务器，1为使用，0为不使用

命令行代码示例：
python main.py --IsUseELG 0 --IsRealTime 0 --modelPath ./model256_20.pth

python main.py --IsUseServer 1 

注：
什么都不写直接运行python main.py，默认运行ELG+VGE模型实时处理视频流：python main.py ==IsUseELG 1 --IsRealTime 1 --modelPath ./model256_20.pth

运行服务器代码需在服务器端同步运行服务器代码，并且在代码里设置相应的ip地址和端口号

按esc可退出程序

'''

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='Your program description')

# 添加参数
parser.add_argument('--IsUseELG', type=int, default=True, help='是否使用ELG模型估计瞳孔中心')
parser.add_argument('--IsRealTime', type=int, default=True, help='是否实时处理视频流，否则先录制视频再处理')
parser.add_argument('--modelPath', type=str, default='./model256_20.pth', help='模型路径')

parser.add_argument('--IsUseServer', type=int, default=False, help='是否使用服务器')



# 解析参数
args = parser.parse_args()

# 使用参数
if args.IsUseServer:
    subprocess.run(['python', 'useServer.py'])
    exit()

if args.IsUseELG:
    if args.IsRealTime:
        RTEV(args.modelPath)
    else:
        VEV(args.modelPath)
else:
    if args.IsRealTime:
        RTV(args.modelPath)
    else:
        VV(args.modelPath)

