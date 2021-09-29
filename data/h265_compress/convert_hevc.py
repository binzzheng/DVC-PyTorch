import os

#class_B  1920x1024
num = 5
video_name = ['BasketballDrive_1920x1080_50.yuv', 'BQTerrace_1920x1080_60.yuv', 'Cactus_1920x1080_50.yuv', 'Kimono1_1920x1080_24.yuv', 'ParkScene_1920x1080_24.yuv']
short = ['BasketballDrive', 'BQTerrace', 'Cactus', 'Kimono1', 'ParkScene']

#class_C  832x448
#num = 4
#video_name = ['BasketballDrill_832x480_50.yuv', 'BQMall_832x480_60.yuv', 'PartyScene_832x480_50.yuv', 'RaceHorses_832x480_30.yuv']
#short = ['BasketballDrill', 'BQMall', 'PartyScene', 'RaceHorses', ]

for i in range(num):
    saveroot = 'images/' + short[i]
    savepath = 'images/' + short[i] + '/im%03d.png'
    if not os.path.exists(saveroot):
        os.makedirs(saveroot)
    print('ffmpeg -y -pix_fmt yuv420p -s 1920x1024 -i ' + 'videos_crop/' + video_name[i] +  ' ' + savepath)
    os.system('ffmpeg -y -pix_fmt yuv420p -s 1920x1024 -i ' + 'videos_crop/' + video_name[i] +  ' ' + savepath)
