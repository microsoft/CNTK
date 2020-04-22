#!/bin/bash
d=${BUILDDIR:-$PWD}
echo "${BUILDDIR:-$PWD}"
for p in ${@##-*}
do
cd "$d"
echo "https://aur.archlinux.org/cgit/aur.git/snapshot/$p.tar.gz" 
curl "https://aur.archlinux.org/cgit/aur.git/snapshot/$p.tar.gz" |tar xz
cd "$p"
sed -i -e 's=git://github.com/falconindy/cower=git+https://github.com/falconindy/cower.git=g' PKGBUILD
echo "makepkg --skippgpcheck ${@##[^\-]*}"
git config --global url."https://".insteadOf git://
makepkg --skippgpcheck ${@##[^\-]*}
echo "$d"
# /usr/sbin/find / -name "$p.*" 
# ls -laR "$d"
# su root -c 'pacman -U $p.tar.xz'
done
