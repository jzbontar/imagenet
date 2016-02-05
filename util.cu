extern "C" {
	#include "lua.h"
	#include "lualib.h"
	#include "lauxlib.h"
}

#include "luaT.h"
#include "THC.h"

#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#define TB 128

THCState* getCutorchState(lua_State* L)
{
	lua_getglobal(L, "cutorch");
	lua_getfield(L, -1, "getState");
	lua_call(L, 0, 1);
	THCState *state = (THCState*) lua_touserdata(L, -1);
	lua_pop(L, 2);
	return state;
}

static const struct luaL_Reg funcs[] = {
	{NULL, NULL}
};

extern "C" int luaopen_libutil(lua_State *L) {
	luaL_openlib(L, "util", funcs, 0);
	return 1;
}
