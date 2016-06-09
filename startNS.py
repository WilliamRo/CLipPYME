import Pyro.naming as naming

def main():
    nss = naming.NameServerStarter()
    nss.start()

if __name__ == "__main__":
    main()