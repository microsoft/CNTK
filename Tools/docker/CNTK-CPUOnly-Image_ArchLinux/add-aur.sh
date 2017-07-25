#!/usr/bin/env bash
# this script sets up unattended aur access via pacaur for a user given as the first argument
set -o pipefail -e

[[ -z "$1" ]] && echo "You must specify a user name" && exit 1
AUR_USER=$1

# create the user
useradd -m $AUR_USER

# set the user's password to blank
echo "${AUR_USER}:" | chpasswd -e

# install devel packages
pacman -S --needed --noprogressbar --noconfirm base-devel

# give the aur user passwordless sudo powers
echo "$AUR_USER      ALL = NOPASSWD: ALL" >> /etc/sudoers

# use all possible cores for subsequent package builds
sed -i 's,#MAKEFLAGS="-j2",MAKEFLAGS="-j$(nproc)",g' /etc/makepkg.conf

# don't compress the packages built here
sed -i "s,PKGEXT='.pkg.tar.xz',PKGEXT='.pkg.tar',g" /etc/makepkg.conf

# install pacaur
su $AUR_USER -c 'mkdir /home/docker/sandground'
su $AUR_USER -c 'cd /home/docker/sandground; bash /usr/sbin/aur.sh -si --noconfirm --needed cower-git pacaur'
echo "step 1"
export editor=vi
export visual=vi
echo "$editor"
echo "$visual"
/usr/sbin/find / -name mysqld.sock
# su $AUR_USER -c 'cd; rm -rf cower pacaur'

# do a pacaur system update
su $AUR_USER -c 'pacaur -Syyua --noprogressbar --noedit --noconfirm'

echo "Packages from the AUR can now be installed like this:"
echo "su $AUR_USER -c 'pacaur -S --needed --noprogressbar --noedit --noconfirm PACKAGE'"
