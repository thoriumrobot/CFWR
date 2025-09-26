using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class PolyCrash {
    void test1(int? integer) {
        ++integer;
    }
}
