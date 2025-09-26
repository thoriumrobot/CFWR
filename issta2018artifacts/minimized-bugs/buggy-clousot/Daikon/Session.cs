using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// simplified version of bug fixed in Daikon commit cc9edee13

abstract class InputStream
{
    public virtual int read(byte[] buf)
    {
        Contract.Ensures(Contract.Result<int>() >= -1 && Contract.Result<int>() <= buf.Length);
        return -1;
    }
}

class Session
{
    public static void do_session(InputStream @is, byte[] buf)
    {
        Contract.Requires(buf != null);
        int pos = @is.read(buf);
        //:: error: argument.type.incompatible
        string actual = Encoding.ASCII.GetString(buf, 0, pos);
    }
}

