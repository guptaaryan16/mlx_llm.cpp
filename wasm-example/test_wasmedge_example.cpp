#include <wasmedge/wasmedge.h>
#include <iostream>

int main(int argc, char **argv)
{
    /* Create the configure context and add the WASI support. */
    /* This step is not necessary unless you need WASI support. */
    


    WasmEdge_ConfigureContext *conf_cxt = WasmEdge_ConfigureCreate();
    WasmEdge_ConfigureAddHostRegistration(conf_cxt, WasmEdge_HostRegistration_Wasi);
    /* The configure and store context to the VM creation can be NULL. */
    WasmEdge_VMContext *vm_cxt = WasmEdge_VMCreate(conf_cxt, nullptr);

    /* The parameters and returns arrays. */
    WasmEdge_Value params[1] = {WasmEdge_ValueGenI32(40)};
    WasmEdge_Value returns[1];
    /* Function name. */
    WasmEdge_String func_name = WasmEdge_StringCreateByCString("fib");
    /* Run the WASM function from file. */
    WasmEdge_Result res = WasmEdge_VMRunWasmFromFile(vm_cxt, argv[1], func_name, params, 1, returns, 1);

    if (WasmEdge_ResultOK(res))
    {
        std::cout << "Get result: " << WasmEdge_ValueGetI32(returns[0]) << std::endl;
    }
    else
    {
        std::cout << "Error message: " << WasmEdge_ResultGetMessage(res) << std::endl;
    }

    /* Resources deallocations. */
    WasmEdge_VMDelete(vm_cxt);
    WasmEdge_ConfigureDelete(conf_cxt);
    WasmEdge_StringDelete(func_name);
    return 0;
}