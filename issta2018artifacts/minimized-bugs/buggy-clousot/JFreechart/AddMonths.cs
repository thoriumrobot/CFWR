using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// This is a simplified version of a bug from JFreeChart that was confirmed by developers.

public class AddMonths
{
    static readonly int [] LAST_DAY_OF_MONTH = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    
    public static void addMonths(int months, int baseYear, int baseMonth) {
        Contract.Requires(baseYear >= 1900 && baseYear <= 9999);
        Contract.Requires(baseMonth >= 1 && baseMonth <= 12);
        int mm = (12 * baseYear + baseMonth + months - 1) % 12 + 1;
        //:: error: argument.type.incompatible
        int lastDayOfMonth = LastDayOfMonth(mm);
    }

    public static int LastDayOfMonth(int month) {
        Contract.Requires(month >= 1 && month <= 12);
        int result = LAST_DAY_OF_MONTH[month];
	    return result;
    }

    [ContractInvariantMethod]
    private void Invariant()
    {
        Contract.Invariant(LAST_DAY_OF_MONTH.Length == 13);
        Contract.Invariant(Contract.ForAll(LAST_DAY_OF_MONTH, i => i >= 0 && i <= 31));
    }
}
