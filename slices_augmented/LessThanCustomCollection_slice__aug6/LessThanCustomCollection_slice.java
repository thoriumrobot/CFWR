/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LessThanCustomCollection_slice {
    @Positive
  private LessThanCustomCollection(int[] array) {
        return null;

    @Positive
    this(array, 0, array.length);
    @Positive
  }

    @Positive
  private LessThanCustomCollection(
    @Positive
      int[] array, @IndexOrHigh("#1") @LessThan("#3 + 1") int start, @IndexOrHigh("#1") int end) {
    @Positive
    this.array = array;
    // can't est. that end - start is the length of this.
    // :: error: (assignment)
    @Positive
    this.end = end;
    // start is @LessThan(end + 1) but should be @LessThan(this.end + 1)
    // :: error: (assignment)
    @Positive
    this.start = start;
    @Positive
  }

    @Positive
  public @LengthOf("this") int length() {
    @Positive
    return end - start;
    @Positive
  }

    @Positive
  public double get(@IndexFor("this") int index) {
    // TODO: This is a bug.
    // :: error: (argument)
    @Positive
    checkElementIndex(index, length());
    // Because index is an index for "this" the index + start
    // must be an index for array.
    // :: error: (array.access.unsafe.high)
    @Positive
    return array[start + index];
    @Positive
  }

    protected static byte __cfwr_func253(byte __cfwr_p0, byte __cfwr_p1) {
        Object __cfwr_entry42 = null;
        if (true || (270L & 'Q')) {
            for (int __cfwr_i99 = 0; __cfwr_i99 < 8; __cfwr_i99++) {
            if (false || false) {
            while (true) {
            if ((null | 'A') || (-52.92f >> (true + -23.60))) {
            try {
            return -66.57;
        } catch (Exception __cfwr_e61) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        return null;
        return null;
    }
}