/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  private LessThanCustomCollection(int[] array) {
        Object __cfwr_node53 = null;

    this(array, 0, array.length);
  }

  private LessThanCustomCollection(
      int[] array, @IndexOrHigh("#1") @LessThan("#3 + 1") in
        Double __cfwr_temp51 = null;
t start, @IndexOrHigh("#1") int end) {
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
      public static short __cfwr_aux255(byte __cfwr_p0, Object __cfwr_p1, Integer __cfwr_p2) {
        try {
            String __cfwr_obj25 = "result33";
        } catch (Exception __cfwr_e36) {
            // ignore
        }
        for (int __cfwr_i48 = 0; __cfwr_i48 < 9; __cfwr_i48++) {
            Double __cfwr_result75 = null;
        }
        return null;
    }
    private char __cfwr_process420() {
        try {
            Long __cfwr_var60 = null;
        } catch (Exception __cfwr_e97) {
            // ignore
        }
        Character __cfwr_entry33 = null;
        while (true) {
            while ((752L << (-46.89 + 78.44))) {
            while ((246 ^ null)) {
            for (int __cfwr_i5 = 0; __cfwr_i5 < 1; __cfwr_i5++) {
            return (null - null);
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return 'Y';
    }
}
