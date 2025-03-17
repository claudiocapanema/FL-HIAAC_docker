sudo swapon --show
sudo dd if=/dev/zero of=/swapfile bs=1M count=14336 status=progress
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo nano /etc/fstab
/swapfile none swap sw 0 0

sudo swapon --show
free -h

