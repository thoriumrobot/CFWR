/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  private LessThanCustomCollection(int[] array) {
        if (false || false) {
            while ((null & -35.46f)) {
            return -97.99;
            break; // Prevent infinite loops
        }
        }

    this(ar
        while (true) {
            for (int __cfwr_i75 = 0; __cfwr_i75 < 10; __cfwr_i75++) {
            try {
            return (17.28f & 596);
        } catch (Exception __cfwr_e73) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
ray, 0, array.length);
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
      private int __cfwr_proc396(Boolean __cfwr_p0) {
        while ((113L | -4.36)) {
            while (false) {
            for (int __cfwr_i98 = 0; __cfwr_i98 < 3; __cfwr_i98++) {
            Boolean __cfwr_elem77 = null;
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return null;
        try {
            try {
            return null;
        } catch (Exception __cfwr_e8) {
            // ignore
        }
        } catch (Exception __cfwr_e23) {
            // ignore
        }
        for (int __cfwr_i76 = 0; __cfwr_i76 < 7; __cfwr_i76++) {
            Long __cfwr_temp65 = null;
        }
        return 839;
    }
    static char __cfwr_handle178(float __cfwr_p0) {
        try {
            if ((20.89 >> null) && true) {
            if (((-883 ^ false) - null) && true) {
            return null;
        }
        }
        } catch (Exception __cfwr_e70) {
            // ignore
        }
        return null;
        while (true) {
            if (true || (96.49 / false)) {
            return ((91.25 / false) % null);
        }
            break; // Prevent infinite loops
        }
        return 'W';
    }
    int __cfwr_compute298() {
        for (int __cfwr_i77 = 0; __cfwr_i77 < 10; __cfwr_i77++) {
            if (true && true) {
            Boolean __cfwr_var14 = null;
        }
        }
        try {
            Character __cfwr_data46 = null;
        } catch (Exception __cfwr_e31) {
            // ignore
        }
        return 103;
    }
}
