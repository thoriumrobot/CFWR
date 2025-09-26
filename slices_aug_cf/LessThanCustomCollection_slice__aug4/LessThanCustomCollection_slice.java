/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
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
      Character __cfwr_process821(Object __cfwr_p0, double __cfwr_p1) {
        return 976;
        try {
            try {
            for (int __cfwr_i6 = 0; __cfwr_i6 < 7; __cfwr_i6++) {
            return null;
        }
        } catch (Exception __cfwr_e40) {
            // ignore
        }
        } catch (Exception __cfwr_e46) {
            // ignore
        }
        try {
            return (true % 36.09);
        } catch (Exception __cfwr_e76) {
            // ignore
        }
        try {
            if (true && true) {
            return null;
        }
        } catch (Exception __cfwr_e6) {
            // ignore
        }
        return null;
    }
    protected static float __cfwr_temp636(Character __cfwr_p0) {
        return null;
        try {
            for (int __cfwr_i74 = 0; __cfwr_i74 < 5; __cfwr_i74++) {
            for (int __cfwr_i33 = 0; __cfwr_i33 < 9; __cfwr_i33++) {
            while ((null << 'Y')) {
            int __cfwr_item67 = -949;
            break; // Prevent infinite loops
        }
        }
        }
        } catch (Exception __cfwr_e11) {
            // ignore
        }
        return ((947 ^ null) / false);
    }
    private short __cfwr_temp368() {
        int __cfwr_data89 = (null / (true & 513));
        return null;
    }
}
