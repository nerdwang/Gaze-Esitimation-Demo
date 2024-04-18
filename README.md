# Gaze-Esitimation-Demo
First of all, please use environment.txt to set up the environment required for running the program.<br><br><br>
Next, download the VGE-Net pretrained model from [this link](https://mega.nz/file/AMxSnDKD#aZrwHfSiPu6-XtUas4Q1HbaePIZOvkyupiTnB2n8upc) and place it in the root directory of the project.<br><br><br>
If you want to run the program locally on your computer : simply execute `python main.py` in the command line. <br><br><br>
If you prefer to offload the neural network computations to a GPU server : <br>
1. First, replace the `server_host` value in `useServer.py` on line 124 with your server's IP address. The default IP address is set to `10086`, but you can change it accordingly. <br>
2. Then, start the server by running `python main.py --IsUseServer 1` to begin listening on port `10086`. <br>
3. Finally, execute `python main.py --IsUseServer 1` on your local machine to establish a connection with the server for data transmission.<br><br><br>

In our development, we leverage insights from both academic research and existing implementations, such as the gaze estimation techniques discussed by Huang et al. (2023) and the GazeML_torch implementation on GitHub.<br><br>
In the end, I want to give big thanks to my bro, Borui. He has put a lot of effort into the deployment of the project and has always been a supportive brother to me. I wish him more and more happiness in his life ahead!

## References

- [Huang, G., Shi, J., Xu, J., Li, J., Chen, S., Du, Y., Zhen, X., & Liu, H. (2023). Gaze Estimation by Attention-Induced Hierarchical Variational Auto-Encoder. IEEE Transactions on Cybernetics, 1-14.](https://doi.org/10.1109/TCYB.2023.3312392)
- [GazeML_torch Implementation on GitHub](https://github.com/J094/GazeML_torch)




https://github.com/nerdwang/Gaze-Esitimation-Demo/assets/129079614/ed077870-79e0-4842-9314-24442e33abf1

