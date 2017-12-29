Summary: Compiler Cache
Name: ccache
Version: 2.3
Release: 1
Group: Development/Languages
License: GPL
URL: http://ccache.samba.org/
Source: ccache-%{version}.tar.gz
BuildRoot: %{_tmppath}/%{name}-%{version}-root

%description
ccache caches gcc output files

%prep
%setup -q

%build
%configure
make

install -d -m 0755 $RPM_BUILD_ROOT%{_bindir}
install -m 0755 ccache $RPM_BUILD_ROOT%{_bindir}
install -d -m 0755 $RPM_BUILD_ROOT%{_mandir}/man1
install -m 0644 ccache.1 $RPM_BUILD_ROOT%{_mandir}/man1

%files
%defattr(-,root,root)
%doc README
%{_mandir}/man1/ccache.1*
%{_bindir}/ccache

%clean
[ "$RPM_BUILD_ROOT" != "/" ] && rm -rf $RPM_BUILD_ROOT

%changelog
* Mon Apr 01 2002 Peter Jones <pjones@redhat.com>
- Created the package
