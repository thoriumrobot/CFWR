/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LessThanCustomCollection_slice {
  private LessThanCustomCollection(int[] array) {
        Character __cfwr_node98 = null;

    this(array, 0, array.length);
  }

  private LessThanCustomCollection(
      int[] array, @IndexOrHigh("#1") @LessThan("#3 + 1") int start, @IndexOrHigh("#1") int end) {
    this.array = array;
    // can't est. that end - start is the length of this.
    // :: error: (assignment)
    this.end = end;
    // start is @LessThan(end + 1) but should be @LessThan(this.end + 1)
    // :: error: (assignment)
    this.start = start;
  }

  @Pure
  public @LengthOf("this") int length() {
    return end - start;
  }

  public double get(@IndexFor("this") int index) {
    // TODO: This is a bug.
    // :: error: (argument)
    checkElementIndex(index, length());
    // Because index is an index for "this" the index + start
    // must be an index for array.
    // :: error: (array.access.unsafe.high)
    return array[start + index];
  }

  public static @NonNegative int checkElementIndex(
      @LessThan("#2") @NonNegative int index, @NonNegative int size) {
    if (index < 0 || index >= size) {
      throw new IndexOutOfBoundsException();
    }
    return index;
  }

  public @IndexOrLow("this") int indexOf(double target) {
    for (int i = start; i < end; i++) {
      if (areEqual(array[i], target)) {
        // Don't know that it is greater than start.
        // :: error: (return)
        return i - start;
      }
    }
    return -1;
  }

    protected static short __cfwr_process434(long __cfwr_p0) {
        for (int __cfwr_i36 = 0; __cfwr_i36 < 8; __cfwr_i36++) {
            try {
            for (int __cfwr_i92 = 0; __cfwr_i92 < 4; __cfwr_i92++) {
            if (false && true) {
            long __cfwr_node2 = (-548 + -158L);
        }
        }
        } catch (Exception __cfwr_e47) {
            // ignore
        }
        }
        return (-24.38f << (null * 'h'));
    }
}