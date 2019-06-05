import configparser


def save():
    with open("config.ini", "w") as configfile:
        config.write(configfile)


config = configparser.ConfigParser()
config.read("config.ini")
config.save = save
