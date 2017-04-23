package = "multiverso"
version = "scm-1"

source = {
    url = "https://github.com/Microsoft/multiverso"
}

description = {
    summary = "Torch binding for multiverso.",
    detailed = [[
        Multiverso is a parameter server framework for distributed machine
        learning. This package can leverage multiple machines and GPUs to
        speed up the torch programs.
    ]],
    homepage = "http://www.dmtk.io",
    license = "MIT"
}

dependencies = {
    "torch >= 7.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
   ]],
   install_command = [[
cd build && $(MAKE) install;
    ]]
}
