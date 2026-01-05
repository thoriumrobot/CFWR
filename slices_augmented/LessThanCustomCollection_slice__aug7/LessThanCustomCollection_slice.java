/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LessThanCustomCollection_slice {
    @Positive
  private LessThanCustomCollection(int[] array) {
        try {
            try {
            for (int __cfwr_i32 = 0; __cfwr_i32 < 5; __cfwr_i32++) {
            for (int __cfwr_i93 = 0; __cfwr_i93 < 5; __cfwr_i93++) {
            try {
            for (int __cfwr_i78 = 0; __cfwr_i78 < 1; __cfwr_i78++) {
            Boolean __cfwr_item61 = null;
        }
        } catch (Exception __cfwr_e36) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e74) {
            // ignore
        }
        } catch (Exception __cfwr_e45) {
            // ignore
        }

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

    public static double __cfwr_func829(double __cfwr_p0) {
        try {
            for (int __cfwr_i49 = 0; __cfwr_i49 < 5; __cfwr_i49++) {
            while (true) {
            if (false && ((false - -120L) * null)) {
            return -777;
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e70) {
            // ignore
        }
        for (int __cfwr_i83 = 0; __cfwr_i83 < 4; __cfwr_i83++) {
            while (false) {
            for (int __cfwr_i25 = 0; __cfwr_i25 < 9; __cfwr_i25++) {
            byte __cfwr_item57 = null;
        }
            break; // Prevent infinite loops
        }
        }
        if (true || (-98.04f + (-726 % 'H'))) {
            try {
            for (int __cfwr_i34 = 0; __cfwr_i34 < 5; __cfwr_i34++) {
            return null;
        }
        } catch (Exception __cfwr_e42) {
            // ignore
        }
        }
        return (402 >> 'I');
    }
}