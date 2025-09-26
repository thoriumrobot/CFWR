/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LessThanCustomCollection_slice {
  private LessThanCustomCollection(int[] array) {
        return null;

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

    float __cfwr_aux767(String __cfwr_p0, Character __cfwr_p1) {
        for (int __cfwr_i99 = 0; __cfwr_i99 < 4; __cfwr_i99++) {
            short __cfwr_var67 = null;
        }
        if (true && (351 << -85.66)) {
            if (false && true) {
            for (int __cfwr_i6 = 0; __cfwr_i6 < 7; __cfwr_i6++) {
            while (false) {
            if (true && false) {
            if (false && false) {
            try {
            try {
            for (int __cfwr_i81 = 0; __cfwr_i81 < 5; __cfwr_i81++) {
            if (true || true) {
            try {
            for (int __cfwr_i24 = 0; __cfwr_i24 < 4; __cfwr_i24++) {
            while ((null << 91.93)) {
            for (int __cfwr_i47 = 0; __cfwr_i47 < 9; __cfwr_i47++) {
            try {
            for (int __cfwr_i63 = 0; __cfwr_i63 < 6; __cfwr_i63++) {
            if (true || true) {
            for (int __cfwr_i28 = 0; __cfwr_i28 < 6; __cfwr_i28++) {
            Float __cfwr_item11 = null;
        }
        }
        }
        } catch (Exception __cfwr_e36) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e74) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e18) {
            // ignore
        }
        } catch (Exception __cfwr_e65) {
            // ignore
        }
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        try {
            float __cfwr_item81 = 66.05f;
        } catch (Exception __cfwr_e33) {
            // ignore
        }
        boolean __cfwr_temp68 = true;
        return 39.61f;
    }
}