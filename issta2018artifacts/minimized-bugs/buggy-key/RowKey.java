// import org.checkerframework.checker.index.qual.*;

public class RowKey { 
    private Underlying underlying;

    /* @ public invariant firstCategoryIndex >= 0 */
    private int firstCategoryIndex;

    /*@ public normal_behavior
@ requires true;
@ ensures \result >= -1;
@*/
    public int getRowIndex(Comparable rowKey) {
	return -1;
    }

        /*@ public normal_behavior
@ requires true;
@ ensures \result >= -1;
@*/
    public int getColumnIndex(Comparable rowKey) {
	return -1;
    }

            /*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
    public Number getValue(Comparable rowKey, Comparable columnKey) {
        int r = getRowIndex(rowKey);
        int c = getColumnIndex(columnKey);
        if (c != -1) {
            //:: error: (assignment.type.incompatible)
            Number result = this.underlying.getValue(r, c + this.firstCategoryIndex);
            return result;
        }
        else {
            throw new UnknownKeyException("Unknown columnKey: " + columnKey);
        }
    }

    interface Underlying {
	        /*@ public normal_behavior
@ requires r >= 0 && c >= 0;
@ ensures true;
@*/
	Number getValue(int r, int c);
    }
    /**
     * An exception that indicates an unknown key value.
     */
    class UnknownKeyException extends IllegalArgumentException {

	/**
	 * Creates a new exception.
	 *
	 * @param message  a message describing the exception.
	 */
	        /*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
	public UnknownKeyException(String message) {
	    super(message);
	}	
    }
}
