import paramiko


def transfer(ssh, file, local):
    # Open an SFTP session
    sftp = ssh.open_sftp()

    def size_str(size):
        if size < 1024:
            return f"{size}B"

        if size < 1024 ** 2:
            return f"{size / 1024:.2f}KB"

        if size < 1024 ** 3:
            return f"{size / 1024 ** 2:.2f}MB"

        return f"{size / 1024 ** 3:.2f}GB"

    # Print progress each 10MB if it's more then 1%
    transferred_before = 0
    percent_before = 0
    def print_progress(transferred, total):
        nonlocal transferred_before, percent_before
        percent = int(transferred / total * 100)

        if (transferred - transferred_before) > (10 * (1024 ** 2)) and percent > percent_before:
            transferred_before = transferred
            percent_before = percent

            print(f"{file}: {size_str(transferred)}\t{percent}%", flush=True)

    # Download the file
    sftp.get(file, local, callback=print_progress)

    # Close the SFTP session and SSH connection
    sftp.close()

if __name__ == "__main__":
    import argparse

    # Get the data from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("hostname", help="Hostname")
    parser.add_argument("username", help="Username")
    parser.add_argument("password", help="Password")
    parser.add_argument("file", help="The file to download")
    parser.add_argument("local", help="The file to download to")

    args = parser.parse_args()

    hostname = args.hostname
    username = args.username
    password = args.password
    file = args.file
    local = args.local

    # Connect to the remote server using SSH
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=hostname, username=username, password=password)

    transfer(ssh, file, local)

    ssh.close()