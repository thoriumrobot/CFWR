/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  private LessThanCustomCollection(int[] array) {
        try {
            return -23.86;
        } catch (Exception __cfwr_e11) {
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
      public Float __cfwr_helper133() {
        Boolean __cfwr_obj61 = null;
        while (true) {
            if (((318 >> true) >> null) && false) {
            try {
            if (false && false) {
            Long __cfwr_entry6 = null;
        }
        } catch (Exception __cfwr_e72) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        while (true) {
            while (true) {
            for (int __cfwr_i44 = 0; __cfwr_i44 < 6; __cfwr_i44++) {
            while (true) {
            try {
            for (int __cfwr_i93 = 0; __cfwr_i93 < 6; __cfwr_i93++) {
            return 951L;
        }
        } catch (Exception __cfwr_e79) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        Float __cfwr_var4 = null;
        return null;
    }
}
