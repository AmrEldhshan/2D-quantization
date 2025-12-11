#include <iostream>

struct SuffixNode
{
    int ind;
    int r1;
    int r2;
};

class SuffixArray
{
private:
    char* text;
    int length;
    SuffixNode* suffixes;

    int my_strlen(const char* s)
    {
        int len = 0;
        while (s[len] != '\0')
        {
            len++;
        }
        return len;
    }

    void my_strcpy(char* dest, const char* src)
    {
        int i = 0;
        while (src[i] != '\0')
        {
            dest[i] = src[i];
            i++;
        }
        dest[i] = '\0';
    }

    template <typename T>
    void copy_arr(const T* src, T* dest, const int &n)
    {
        for (int i = 0; i < n; ++i)
            dest[i] = src[i];
    }

    template <typename T>
    static bool default_compare(const T& a, const T& b)
    {
        return a < b;
    }

    template <typename T, bool (*CMP)(const T&, const T&) = default_compare<T>>
    void sort(T arr[], const int &n)
    {
        if (n < 2)
            return;

        int n1 = n >> 1, n2 = n - n1;

        T left[n1], right[n2];

        copy_arr(arr, left, n1);
        copy_arr(arr + n1, right, n2);

        sort<T, CMP>(left, n1);
        sort<T, CMP>(right, n2);

        int i = 0, j = 0, k = 0;
        while (i < n1 && j < n2)
        {
            if (CMP(right[j], left[i]))
            {
                arr[k++] = right[j++];
            } else {
                arr[k++] = left[i++];
            }
        }

        while (i < n1) arr[k++] = left[i++];
        while (j < n2) arr[k++] = right[j++];
    }

    // --- Comparators for SuffixNode ---

    static bool compareByRank(const SuffixNode& a, const SuffixNode& b)
    {
        if (a.r1 != b.r1)
        {
            return a.r1 < b.r1;
        }
        return a.r2 < b.r2;
    }

    static bool compareByIndex(const SuffixNode& a, const SuffixNode& b)
    {
        return a.ind < b.ind;
    }

public:
    SuffixArray(const char* str)
    {
        length = my_strlen(str);
        text = new char[length + 1];
        my_strcpy(text, str);
        suffixes = new SuffixNode[length];
    }

    ~SuffixArray()
    {
        delete[] text;
        delete[] suffixes;
    }

    void ConstructUsingPrefixDoubling()
    {
        for (int i = 0; i < length; i++)
        {
            suffixes[i].ind = i;
            suffixes[i].r1 = text[i];
            suffixes[i].r2 = (i + 1 < length) ? text[i + 1] : -1;
        }

        for (int k = 1; k < length; k *= 2)
        {
            // Sort by <r1, r2>
            sort<SuffixNode, compareByRank>(suffixes, length);

            int* newRanks = new int[length];
            int rank = 0;
            newRanks[0] = 0;

            for (int i = 1; i < length; i++)
            {
                if (suffixes[i].r1 == suffixes[i - 1].r1 &&
                    suffixes[i].r2 == suffixes[i - 1].r2)
                    {
                    newRanks[i] = rank;
                }
                else
                {
                    rank++;
                    newRanks[i] = rank;
                }
            }

            for (int i = 0; i < length; i++)
            {
                suffixes[i].r1 = newRanks[i];
            }

            delete[] newRanks;

            // Sort by index to prepare for next iteration's r2 update
            sort<SuffixNode, compareByIndex>(suffixes, length);

            for (int i = 0; i < length; i++)
            {
                int nextIndex = suffixes[i].ind + k;
                if (nextIndex < length)
                {
                    suffixes[i].r2 = suffixes[nextIndex].r1;
                } else
                {
                    suffixes[i].r2 = -1;
                }
            }
        }

        // Final Sort by Rank
        sort<SuffixNode, compareByRank>(suffixes, length);
    }

    void Print()
    {
        for (int i = 0; i < length; i++)
        {
            printf("%d", suffixes[i].ind) ;
            if (i < length - 1)
            {
               printf(" ");
            }
        }
        printf("\n");
    }

};


int main()
{
    SuffixArray t("ACGACTACGATAAC$");

    t.ConstructUsingPrefixDoubling();

    t.Print(); // Prints:  14 11 12  0  6  3  9 13  1  7  4  2  8 10  5

    return 0;
}
