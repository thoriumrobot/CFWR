using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class ParsingBug {
    void test() {
        String[] saOrig = new String[] {"foo", "bar"};
        Object o1 = do_things((Object) saOrig);
    }

    Object do_things(Object o) {
        return o;
    }
}
