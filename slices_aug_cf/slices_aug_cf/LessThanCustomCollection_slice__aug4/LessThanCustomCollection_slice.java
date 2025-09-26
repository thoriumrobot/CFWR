/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  private LessThanCustomCollection(int[] array) {
        for (int __cfwr_i80 = 0; __cfwr_i80 < 10; __cfwr_i80++) {
            try {
            return (-357 * null);
        } catch (Exception __cfwr_e75) {
            // ignore
        }
        }

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
      private static Double __cfwr_temp385(double __cfwr_p0) {
        return null;
        while ((933L << -74.40)) {
            try {
            return null;
        } catch (Exception __cfwr_e44) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return null;
    }
    public static float __cfwr_util319(short __cfwr_p0) {
        Long __cfwr_result62 = null;
        return -9.58f;
    }
}
