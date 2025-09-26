using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;


public class SameLenWithObjects {

    protected class SimpleCollection {
        protected internal Object[] var_infos;
    }

    protected class Invocation1 {
        protected SimpleCollection sc;
        protected Object[] vals1;

        void format1() {
            for (int j = 0; j < vals1.Length; j++) {
                Console.WriteLine(sc.var_infos[j]);
            }
        }

        [ContractInvariantMethod]
        private void Invariant()
        {
            Contract.Invariant(vals1.Length == vals1.Length && vals1.Length == this.sc.var_infos.Length);
        }
    }
}
