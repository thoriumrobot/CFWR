/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LessThanCustomCollection_slice {
    @Positive
  private LessThanCustomCollection(int[] array) {
        Float __cfwr_elem98 = null;

    @Positive
    this(array, 0, array.length);
    @Positive
  }

    @Positive
  private LessThanCustomCollection(
    @Positive
      int[] array, @IndexOrHigh("#1") @LessThan("#3 + 1") int
        while (false) {
            byte __cfwr_result96 = null;
            break; // Prevent infinite loops
        }
 start, @IndexOrHigh("#1") int end) {
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

    private double __cfwr_handle895(Boolean __cfwr_p0, char __cfwr_p1, char __cfwr_p2) {
        while ((61.91f ^ (null ^ -119L))) {
            while (true) {
            for (int __cfwr_i61 = 0; __cfwr_i61 < 2; __cfwr_i61++) {
            try {
            while ((725 * 957L)) {
            if (false || false) {
            long __cfwr_item22 = 226L;
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e32) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return (null | true);
    }
    protected static byte __cfwr_calc227(char __cfwr_p0, double __cfwr_p1, byte __cfwr_p2) {
        try {
            while (true) {
            float __cfwr_elem62 = 62.93f;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e44) {
            // ignore
        }
        for (int __cfwr_i54 = 0; __cfwr_i54 < 4; __cfwr_i54++) {
            while (false) {
            while (false) {
            for (int __cfwr_i7 = 0; __cfwr_i7 < 4; __cfwr_i7++) {
            if ((432 ^ (null ^ -627L)) && true) {
            int __cfwr_elem90 = ((77.85f + 9.62f) >> (null / -764L));
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        try {
            return null;
        } catch (Exception __cfwr_e53) {
            // ignore
        }
        double __cfwr_entry44 = 78.05;
        return null;
    }
}