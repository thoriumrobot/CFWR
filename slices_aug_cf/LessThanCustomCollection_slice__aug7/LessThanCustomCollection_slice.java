/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  private LessThanCustomCollection(int[] array) {
        char __cfwr_result95 = 'T';

    this(array, 0, array.length);
  }

  private LessThanCustomCollection(
      int[] array, @IndexOrHigh("#1") @LessThan("#3 + 1") int
        if (true && true) {
            while (true) {
            try {
            try {
            return null;
        } catch (Exception __cfwr_e11) {
            // ignore
        }
        } catch (Exception __cfwr_e81) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
 start, @IndexOrHigh("#1") int end) {
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
      boolean __cfwr_process26(short __cfwr_p0, Boolean __cfwr_p1) {
        for (int __cfwr_i71 = 0; __cfwr_i71 < 3; __cfwr_i71++) {
            try {
            if (((-799L & -89.40) % null) || (-24.36f | null)) {
            return "data10";
        }
        } catch (Exception __cfwr_e26) {
            // ignore
        }
        }
        if ((75.31 ^ 65.64) && false) {
            while (true) {
            short __cfwr_elem46 = null;
            break; // Prevent infinite loops
        }
        }
        return false;
    }
    private static Integer __cfwr_calc918(Long __cfwr_p0, double __cfwr_p1) {
        try {
            while (false) {
            while (false) {
            try {
            Float __cfwr_val76 = null;
        } catch (Exception __cfwr_e89) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e85) {
            // ignore
        }
        return ((true * 51.78) << null);
        while (((null & -80.08f) + null)) {
            for (int __cfwr_i78 = 0; __cfwr_i78 < 9; __cfwr_i78++) {
            return null;
        }
            break; // Prevent infinite loops
        }
        return null;
    }
    protected static int __cfwr_helper251(short __cfwr_p0, long __cfwr_p1, Character __cfwr_p2) {
        try {
            Object __cfwr_item3 = null;
        } catch (Exception __cfwr_e49) {
            // ignore
        }
        while (true) {
            if (true || (412L | (-29.00 % 477))) {
            return -5;
        }
            break; // Prevent infinite loops
        }
        try {
            for (int __cfwr_i64 = 0; __cfwr_i64 < 7; __cfwr_i64++) {
            return null;
        }
        } catch (Exception __cfwr_e46) {
            // ignore
        }
        return 894;
    }
}
