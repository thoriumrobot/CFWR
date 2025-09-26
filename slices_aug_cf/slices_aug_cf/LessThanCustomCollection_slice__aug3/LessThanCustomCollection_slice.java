/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  private LessThanCustomCollection(int[] array) {
        try {
            Long __cfwr_node65 = null;
        } catch (Exception __cfwr_e65) {
            // ignore
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
      boolean __cfwr_func423(double __cfwr_p0, boolean __cfwr_p1) {
        Float __cfwr_item65 = null;
        char __cfwr_var14 = '0';
        for (int __cfwr_i23 = 0; __cfwr_i23 < 3; __cfwr_i23++) {
            try {
            return null;
        } catch (Exception __cfwr_e50) {
            // ignore
        }
        }
        return false;
    }
    private static Float __cfwr_process76(int __cfwr_p0, String __cfwr_p1) {
        return null;
        for (int __cfwr_i88 = 0; __cfwr_i88 < 9; __cfwr_i88++) {
            long __cfwr_item58 = -720L;
        }
        try {
            Boolean __cfwr_data35 = null;
        } catch (Exception __cfwr_e18) {
            // ignore
        }
        return null;
    }
}
