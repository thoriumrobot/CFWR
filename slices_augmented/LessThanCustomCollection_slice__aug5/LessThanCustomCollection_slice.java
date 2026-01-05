/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LessThanCustomCollection_slice {
    @Positive
  private LessThanCustomCollection(int[] array) {
        if (true && ((false + null) % null)) {
            for (int __cfwr_i86 = 0; __cfwr_i86 < 4; __cfwr_i86++) {
            for (int __cfwr_i64 = 0; __cfwr_i64 < 7; __cfwr_i64++) {
            Long __cfwr_result8 = null;
        }
        }
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

    public static String __cfwr_calc130(float __cfwr_p0, Integer __cfwr_p1, float __cfwr_p2) {
        if ((null - null) || true) {
            byte __cfwr_entry11 = null;
        }
        Character __cfwr_node99 = null;
        while ((true | 'l')) {
            if (true && true) {
            for (int __cfwr_i18 = 0; __cfwr_i18 < 2; __cfwr_i18++) {
            for (int __cfwr_i92 = 0; __cfwr_i92 < 1; __cfwr_i92++) {
            try {
            float __cfwr_elem40 = 64.89f;
        } catch (Exception __cfwr_e68) {
            // ignore
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
        return "data55";
    }
}