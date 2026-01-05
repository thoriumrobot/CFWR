/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LessThanCustomCollection_slice {
    @Positive
  private LessThanCustomCollection(int[] array) {
        if ((367L >> 404L) || true) {
            if (true && true) {
            return null;
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

    public short __cfwr_temp213(Boolean __cfwr_p0, Float __cfwr_p1, short __cfwr_p2) {
        return null;
        while (((-301 - null) >> null)) {
            try {
            if (false && false) {
            while (false) {
            for (int __cfwr_i43 = 0; __cfwr_i43 < 7; __cfwr_i43++) {
            for (int __cfwr_i87 = 0; __cfwr_i87 < 9; __cfwr_i87++) {
            try {
            try {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e26) {
            // ignore
        }
        } catch (Exception __cfwr_e97) {
            // ignore
        }
        }
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e33) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return null;
    }
    char __cfwr_aux198(Boolean __cfwr_p0, double __cfwr_p1, Boolean __cfwr_p2) {
        for (int __cfwr_i1 = 0; __cfwr_i1 < 4; __cfwr_i1++) {
            if (false && false) {
            if (false && false) {
            while (('W' + (589 >> -55.73f))) {
            short __cfwr_entry27 = null;
            break; // Prevent infinite loops
        }
        }
        }
        }
        float __cfwr_val19 = 39.92f;
        if (true || false) {
            return false;
        }
        return '2';
    }
}