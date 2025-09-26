using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// simplified version of bug in plume-lib fixed by b5df093053146bcf32f485eea974e70cc854b407

public class OptionsToHtml
{
    public void optionsToHtml(bool includeClassDoc, params object[] root_classes)
    {
        if (includeClassDoc)
        {
            //:: error: (array.access.unsafe.high.constant)
            javadocToHtml(root_classes[0]);
        }
    }

    public void javadocToHtml(object o) { }
}
