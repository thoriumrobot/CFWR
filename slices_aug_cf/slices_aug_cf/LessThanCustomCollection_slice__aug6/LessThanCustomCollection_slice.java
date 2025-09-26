/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  private LessThanCustomCollection(int[] array) {
        for (int __cfwr_i24 = 0; __cfwr_i24 < 8; __cfwr_i24++) {
            for (int __cfwr_i57 = 0; __cfwr_i57 < 2; __cfwr_i57++) {
            for (int __cfwr_i67 = 0; __
        for (int __cfwr_i71 = 0; __cfwr_i71 < 7; __cfwr_i71++) {
            while (false) {
            try {
            if (true || ((46.92f * 476) * (true * 98.63f))) {
            float __cfwr_val20 = 34.37f;
        }
        } catch (Exception __cfwr_e74) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
cfwr_i67 < 10; __cfwr_i67++) {
            char __cfwr_temp45 = ((null >> true) - 'U');
        }
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
      static Long __cfwr_handle818() {
        if (true && ('c' - (207L >> 96.74))) {
            for (int __cfwr_i67 = 0; __cfwr_i67 < 7; __cfwr_i67++) {
            byte __cfwr_data13 = (408 & (404L >> -31.71));
        }
        }
        return null;
        try {
            for (int __cfwr_i78 = 0; __cfwr_i78 < 7; __cfwr_i78++) {
            if (false || (null % 693L)) {
            if (true && ((null & 423) ^ null)) {
            return 615;
        }
        }
        }
        } catch (Exception __cfwr_e59) {
            // ignore
        }
        while (true) {
            if (true || (null % false)) {
            while (true) {
            long __cfwr_val18 = 773L;
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
