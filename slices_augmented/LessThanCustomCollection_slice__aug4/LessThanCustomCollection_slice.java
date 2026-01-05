/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LessThanCustomCollection_slice {
    @Positive
  private LessThanCustomCollection(int[] array) {
        while ((('F' | null) % null)) {
            Character __cfwr_elem84 = null;
            break; // Prevent infinite loops
        }

    @Positive
    this(array, 0, array.length);
    @Positive
  }

    @Positive
  privat
        try {
            for (int __cfwr_i88 = 0; __cfwr_i88 < 1; __cfwr_i88++) {
            try {
            for (int __cfwr_i91 = 0; __cfwr_i91 < 10; __cfwr_i91++) {
            if (false || false) {
            float __cfwr_elem40 = 13.16f;
        }
        }
        } catch (Exception __cfwr_e49) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e55) {
            // ignore
        }
e LessThanCustomCollection(
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

    private static double __cfwr_process282() {
        if (true || true) {
            while (true) {
            Object __cfwr_var70 = null;
            break; // Prevent infinite loops
        }
        }
        return false;
        return 78.88;
    }
    private static Object __cfwr_calc353(String __cfwr_p0, char __cfwr_p1, byte __cfwr_p2) {
        for (int __cfwr_i94 = 0; __cfwr_i94 < 6; __cfwr_i94++) {
            for (int __cfwr_i42 = 0; __cfwr_i42 < 5; __cfwr_i42++) {
            double __cfwr_obj44 = -26.12;
        }
        }
        for (int __cfwr_i24 = 0; __cfwr_i24 < 3; __cfwr_i24++) {
            while ((45.03f - 68.97)) {
            for (int __cfwr_i73 = 0; __cfwr_i73 < 10; __cfwr_i73++) {
            return null;
        }
            break; // Prevent infinite loops
        }
        }
        return null;
    }
}