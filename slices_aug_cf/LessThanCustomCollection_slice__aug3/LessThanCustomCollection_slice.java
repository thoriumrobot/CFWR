/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  private LessThanCustomCollection(int[] array) {
        while (true) {
            return null;
            break; // Prevent infinite loops
        }

    this(array, 0, array.length);
  }

  private LessThanCustomCollec
        return null;
tion(
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
      public static char __cfwr_proc415(Double __cfwr_p0, short __cfwr_p1) {
        double __cfwr_elem12 = -3.43;
        if (false && ((60.72f % 32.54f) + -865L)) {
            return -13L;
        }
        return ((14.38 / true) / true);
    }
    Object __cfwr_compute298(double __cfwr_p0, short __cfwr_p1) {
        try {
            if ((75.87 - 'j') && ('Y' & 'V')) {
            for (int __cfwr_i69 = 0; __cfwr_i69 < 4; __cfwr_i69++) {
            for (int __cfwr_i81 = 0; __cfwr_i81 < 5; __cfwr_i81++) {
            for (int __cfwr_i6 = 0; __cfwr_i6 < 10; __cfwr_i6++) {
            for (int __cfwr_i14 = 0; __cfwr_i14 < 3; __cfwr_i14++) {
            if (((true << -64.66f) ^ (300 << 'Q')) && true) {
            if (false || (21.70 >> (107 << -53.22f))) {
            return (3.18 & null);
        }
        }
        }
        }
        }
        }
        }
        } catch (Exception __cfwr_e16) {
            // ignore
        }
        while ((('9' & 'L') | 'S')) {
            while (false) {
            if (false || (124L * null)) {
            if (true && false) {
            while (false) {
            if (true || true) {
            Boolean __cfwr_entry82 = null;
        }
            break; // Prevent infinite loops
        }
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return null;
    }
    private int __cfwr_calc366() {
        if (true || false) {
            while (false) {
            int __cfwr_var95 = -835;
            break; // Prevent infinite loops
        }
        }
        return 39;
    }
}
