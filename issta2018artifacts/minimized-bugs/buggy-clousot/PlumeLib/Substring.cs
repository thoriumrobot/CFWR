using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

#if false
// simplified version of plume-lib bug fixed in e813d29

// I'm not sure we should include this. This isn't really an indexing bug - the
// indexing is safe - but rather a bug in what we want this to mean. We only
// noticed it because of the Index Checker, but I don't think it would be fair
// to compare on this bug. And, the Index Checker reports no errors!


public class Substring
{
    public void write(string s, int off, int len) {
        //TODO: what should be here?
        s.Substring(off, len);
    }
}

#endif