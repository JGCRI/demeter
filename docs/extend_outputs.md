# How to extend Demeter to output a new format:

1. After cloning Demeter, create a branch with an appropriate title and check it out on your local machine.
2. Add a new parameter to the input config file.
![config_file](https://github.com/IMMM-SFA/demeter/blob/master/img/extend_config.png "extend config file")
3. In demeter/demeter/config_reader.py add your parameter configuration in the ReadConfig and ReadConfigShuffle classes in the space provided.
![config_reader](https://github.com/IMMM-SFA/demeter/blob/master/img/extend_config_reader.png "extend config reader")
4. Add your function to demeter/demeter/demeter_io/writer.py 
5. Parameterize and call your function in the output method of the ProcessStep class here:  demeter/demeter/process.py
![process](https://github.com/IMMM-SFA/demeter/blob/master/img/extend_process.png "extend process")
6. Make sure it works!
7. Submit a pull request and the admin will review your work and integrate the output format into Demeter!
    