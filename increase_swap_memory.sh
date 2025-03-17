sudo swapon --show
sudo swapoff -a
sudo rm /swapfile
sudo dd if=/dev/zero of=/swapfile bs=1M count=32768
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo nano /etc/fstab
/swapfile none swap sw 0 0

sudo swapon --show
free -h

