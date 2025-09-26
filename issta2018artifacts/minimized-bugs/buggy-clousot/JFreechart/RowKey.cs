using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
public class Number { }
public interface Comparable { }

public class RowKey
{

    protected Underlying underlying;
    protected int firstCategoryIndex;

    [ContractInvariantMethod]
    private void Invariant()
    {
        Contract.Invariant(firstCategoryIndex >= 0);
    }

    public int getRowIndex(Comparable rowKey)
    {
        Contract.Ensures(Contract.Result<int>() >= -1);
        return -1;
    }

    public int getColumnIndex(Comparable rowKey)
    {
        Contract.Ensures(Contract.Result<int>() >= -1);
        return -1;
    }

    public Number getValue(Comparable rowKey, Comparable columnKey)
    {
        int r = getRowIndex(rowKey);
        int c = getColumnIndex(columnKey);
        if (c != -1)
        {
            //:: error: (assignment.type.incompatible)
            Number result = this.underlying.getValue(r, c + this.firstCategoryIndex);
            return result;
        }
        else
        {
            throw new UnknownKeyException("Unknown columnKey: " + columnKey);
        }
    }

    [ContractClass(typeof(UnderlyingContract))]
    protected interface Underlying
    {
        Number getValue(int r, int c);
    }
    [ContractClassFor(typeof(Underlying))]
    abstract class UnderlyingContract : Underlying
    {
        Number Underlying.getValue(int r, int c)
        {
            Contract.Requires(r >= 0);
            Contract.Requires(c >= 0);
            return default(Number);
        }
    }
    /**
     * An exception that indicates an unknown key value.
     */
    class UnknownKeyException : ArgumentException
    {

    /**
	 * Creates a new exception.
	 *
	 * @param message  a message describing the exception.
	 */
    public UnknownKeyException(String message):base(message)
    {
        
    }
}
}