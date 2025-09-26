using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

public class FileCompiler
{

    string [] compiler;
    
    public FileCompiler(List<string> compiler)
    {
        //:: error: assignment.type.incompatible
        this.compiler = compiler.ToArray();
    }

    [ContractInvariantMethod]
    private void Invariant()
    {
        Contract.Invariant(compiler.Length >= 1);
    }
}