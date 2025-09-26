/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  private LessThanCustomCollection(int[] array) {
        while ((278 << -11L)) {
            Character __cfwr_entry3 = null;
            break; // Prevent infinite loops
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
      private boolean __cfwr_handle58() {
        float __cfwr_elem62 = -47.96f;
        return 85.38f;
        if (((null >> null) >> 275L) && (28.72f | false)) {
            while ((true ^ null)) {
            return null;
            break; // Prevent infinite loops
        }
        }
        return false;
    }
    public static boolean __cfwr_compute619() {
        while (false) {
            while ((null << 86.57f)) {
            if (false || (491L - -37.30f)) {
            while (false) {
            while (false) {
            for (int __cfwr_i29 = 0; __cfwr_i29 < 2; __cfwr_i29++) {
            if ((-841L << (95.61 << 24.54f)) && false) {
            Object __cfwr_var88 = null;
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return false;
    }
}
