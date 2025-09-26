using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

public class ItemNum
{
    public object getPreviousEntry(List<object> matching_entries, int? item_num)
    {
        if (item_num != null)
        {
            //:: error: argument.type.incompatible
            return matching_entries[item_num.Value - 1];
        }
        return null;
    }
}