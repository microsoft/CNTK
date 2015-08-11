// main.cpp -- main function for testing config parsing

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "ConfigEvaluator.h"

using namespace Microsoft::MSR::CNTK;

#ifndef let
#define let const auto
#endif

int wmain(int /*argc*/, wchar_t* /*argv*/[])
{
    // there is record of parameters
    // user wants to get a parameter
    // double x = config->GetParam("name", 0.0);
    try
    {
        //let parserTest = L"a=1\na1_=13;b=2 // cmt\ndo = new PrintAction [message='hello'];do1=(print\n:train:eval) ; x = array[1..13] (i=>1+i*print.message==13*42) ; print = new PrintAction [ message = 'Hello World' ]";
        let parserTest1 = L"do3 = new LearnableParameter [ inDim=13; outDim=42 ] * new InputValue [ ] + new LearnableParameter [ outDim=42 ]\n"
            L"do2 = array [1..10] (i=>i*i) ;"
            L"do = new PrintAction [ what = 'abc' ] ;"
            L"do5 = new PrintAction [ what = new StringFunction [ x = 13 ; y = 42 ; what = 'format' ; how = '.2' ; arg = x*y ] ] ;"
            L"do4 = new PrintAction [ what = \"new StringFunction [ what = 'format' ; how = '.2' ; arg = '13 > 42' ]\" ] ;"
            L"do1 = new PrintAction [ what = if 13 > 42 || 12 > 1 then 'Hello World' + \"!\" else 'Oops?']";
        let parserTest2 = L"i2s(i) = new StringFunction [ what = 'format' ; arg = i ; how = '.2' ] ; print(s) = new PrintAction [ what = s ] ; do = print('result=' + i2s((( [ v = (i => i + delta) ].v(5)))+13)) ; delta = 42 ";
        let parserTest3 = L"do = new PrintAction [ what = val ] ; val=1+2*3; text = 'hello'+' world' ";
        let parserTest4 = L"do = new PrintAction [ what = new StringFunction [ what = 'format' ; arg = (13:(fortytwo:1):100) ; how = '' ] ];fortytwo=42 ";
        let parserTest5 = L"do = new PrintAction [ what = val ] ; val=13:[a='a';b=42]:14; arr = array [1..10] (i => 2*i) ";
        let parserTest6 = L"do = new PrintAction [ what = arg ] ; N = 5 ; arr = array [1..N] (i => if i < N then arr[i+1]*i else N) ; arg = arr ";
        let parserTest7 = L"do = new PrintAction [ what = val ] ; val = [ v = (i => i + offset) ].v(42) ; offset = 13 ";
        let parserTest8 = L"Parameters(O,I) = new ComputationNode [ class = 'LearnableParameter'; outDim=O; inDim=I ] \n"
                          L"do = new PrintAction [ what = val ] \n"
                          L"val = new NDLNetwork [\n"
                          L"  A = Parameters(13,42) ; B = A*A+A ; outZ = B*B+A-A \n"
                          L"]\n";
        parserTest1; parserTest2; parserTest3; parserTest4; parserTest5; parserTest6; parserTest7; parserTest8;
        let parserTest = parserTest3;
        let expr = ParseConfigString(parserTest);
        //expr->Dump();
        Do(expr);
        //ParseConfigFile(L"c:/me/test.txt")->Dump();
    }
    catch (const ConfigError & err)
    {
        err.PrintError();
    }
    return EXIT_SUCCESS;
}
