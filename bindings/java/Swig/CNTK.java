package com.microsoft.CNTK;

public final class CNTK {

    public static void init(){
        CNTKNativeUtils.loadAllLibraries();
    }

    private CNTK() {}

    static {
        init();
    }
}
