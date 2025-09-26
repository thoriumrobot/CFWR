/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  private LessThanCustomCollection(int[] array) {
        if ((-4.76 - (386L + false)) || (-80.38 & null)) {
            return null;
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
      private char __cfwr_compute463() {
        for (int __cfwr_i95 = 0; __cfwr_i95 < 3; __cfwr_i95++) {
            while (true) {
            while (true) {
            while ((60.06 / 722)) {
            if (true && true) {
            return null;
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        return 'f';
    }
    private static byte __cfwr_helper487(Float __cfwr_p0) {
        Double __cfwr_elem15 = null;
        try {
            while (false) {
            try {
            Double __cfwr_item52 = null;
        } catch (Exception __cfwr_e13) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e97) {
            // ignore
        }
        for (int __cfwr_i9 = 0; __cfwr_i9 < 7; __cfwr_i9++) {
            try {
            if (false || (82.16f ^ -72.39)) {
            boolean __cfwr_data46 = (null >> -686);
        }
        } catch (Exception __cfwr_e59) {
            // ignore
        }
        }
        if ((null | 24.21f) || (null ^ -428L)) {
            Integer __cfwr_result70 = null;
        }
        return null;
    }
}
